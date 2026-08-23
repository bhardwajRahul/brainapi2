from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from search.dataset import split_corpus
from search.finetune_esci_4class import CLASS_GAINS, rank_doc_ids, weighted_scores
from search.local_dense import evaluate_dense
from search.metrics import aggregate_query_metrics
from search.rank_pool import doc_texts
from search.rank_pool_4class import PredictFn, load_4class_predict

DEFAULT_MODEL = Path("data/models/esci-minilm-l12-4class-nowt-e2")
PROTOCOL = "exhaustive-catalog"


def run_exhaustive_ce(
    rows: list[dict[str, Any]],
    *,
    predict: PredictFn,
    model_name: str,
    dataset_name: str,
    k: int = 50,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    brain_id: str | None = None,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    texts = doc_texts(docs)
    doc_ids = [str(doc.get("doc_id") or "") for doc in docs if doc.get("doc_id")]
    cut = max(1, min(int(k), len(doc_ids) or 1))
    ranked_by_qid: dict[str, list[str]] = {}
    per_q_ms: list[float] = []
    missing_text = 0
    for index, query in enumerate(queries):
        q_text = str(query.get("query") or "")
        pairs: list[tuple[str, str]] = []
        for doc_id in doc_ids:
            text = texts.get(doc_id) or ""
            if not text:
                missing_text += 1
            pairs.append((q_text, text))
        started = time.perf_counter()
        probs = predict(pairs) if pairs else []
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_q_ms.append(elapsed_ms)
        scores = weighted_scores(probs, CLASS_GAINS) if pairs else []
        if len(scores) < len(doc_ids):
            scores = list(scores) + [0.0] * (len(doc_ids) - len(scores))
        ranked = rank_doc_ids(doc_ids, scores)[:cut]
        qid = str(query.get("qid") or "")
        ranked_by_qid[qid] = ranked
        print(f"exhaustive {index + 1}/{len(queries)} {qid} {elapsed_ms:.0f}ms", flush=True)
    encode_ms = sum(per_q_ms)
    metrics, per_query = evaluate_dense(
        ranked_by_qid,
        queries,
        ks=ks,
        encode_ms=encode_ms,
    )
    if not metrics and per_query:
        metrics = aggregate_query_metrics(per_query, ks=ks)
    return {
        "status": "ok" if per_query else "failed",
        "brain_id": brain_id or "harness-local-exhaustive",
        "dataset": dataset_name,
        "fusion": "none",
        "rerank": "4class-weighted-ce",
        "channels": ["exhaustive-4class"],
        "expand": "none",
        "k": cut,
        "ks": list(ks),
        "n_docs": len(docs),
        "n_queries": len(per_query),
        "n_docs_mapped": len(texts),
        "skip_enrichment": True,
        "ingest_graph": False,
        "skip_ingest": True,
        "rank_pool": False,
        "rank_pool_ce": False,
        "ce_model": model_name,
        "ce_missing_text": missing_text,
        "protocol": PROTOCOL,
        "ingest": {
            "status": "completed",
            "n_docs": len(docs),
            "tasks": [],
            "reused": True,
        },
        "graph_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "interaction_ingest": {"status": "skipped", "n_triples": 0, "tasks": []},
        "event_probe": None,
        "search_error": None,
        "metrics": metrics,
        "queries": per_query,
    }


def load_predict(model_dir: Path, *, max_length: int = 192) -> tuple[PredictFn, str]:
    return load_4class_predict(model_dir, max_length=max_length)
