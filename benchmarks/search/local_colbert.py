from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from search.dataset import split_corpus
from search.local_dense import evaluate_dense
from search.pool_first_stage import rank_docs

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "search-colbert"
BRAIN_ID = "harness-local-colbert"


def _plugin():
    if str(PLUGIN_DIR) not in sys.path:
        sys.path.insert(0, str(PLUGIN_DIR))
    from encode import encode_tokens, model_name
    from index import encodings, index_chunks, reset, retrieve

    return encode_tokens, model_name, encodings, index_chunks, reset, retrieve


def _maxsim_scores(
    query_toks: list[list[float]],
    docs: dict[str, list[list[float]]],
) -> dict[str, float]:
    import numpy as np

    ids = list(docs)
    if not ids:
        return {}
    mats = [np.asarray(docs[doc_id], dtype=np.float32) for doc_id in ids]
    q = np.asarray(query_toks, dtype=np.float32)
    if q.ndim != 2 or q.size == 0:
        return {doc_id: 0.0 for doc_id in ids}
    dim = int(q.shape[1])
    max_l = max((int(mat.shape[0]) if mat.ndim == 2 and mat.size else 0) for mat in mats)
    max_l = max(1, max_l)
    stacked = np.zeros((len(mats), max_l, dim), dtype=np.float32)
    mask = np.zeros((len(mats), max_l), dtype=bool)
    for index, mat in enumerate(mats):
        if mat.ndim != 2 or mat.size == 0:
            continue
        width = min(int(mat.shape[1]), dim)
        length = min(int(mat.shape[0]), max_l)
        stacked[index, :length, :width] = mat[:length, :width]
        mask[index, :length] = True
    qn = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    dn = stacked / np.clip(np.linalg.norm(stacked, axis=2, keepdims=True), 1e-12, None)
    dn[~mask] = 0.0
    sims = np.einsum("qd,nld->qnl", qn, dn)
    sims[:, ~mask] = -1e9
    scores = sims.max(axis=2).sum(axis=0)
    return {ids[index]: float(scores[index]) for index in range(len(ids))}


def retrieve_colbert(
    docs: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    k: int = 50,
    brain_id: str = BRAIN_ID,
) -> tuple[dict[str, list[str]], float, str]:
    encode_tokens, model_name, encodings, index_chunks, reset, retrieve = _plugin()
    reset(brain_id)
    chunks: list[dict[str, str]] = []
    doc_ids: list[str] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        if not doc_id:
            continue
        doc_ids.append(doc_id)
        chunks.append(
            {"id": doc_id, "text": str(doc.get("text") or doc.get("title") or "")}
        )
    batch = 64
    started = time.perf_counter()
    for offset in range(0, len(chunks), batch):
        index_chunks(
            brain_id,
            chunks[offset : offset + batch],
            replace=(offset == 0),
        )
        print(
            f"colbert index {min(offset + batch, len(chunks))}/{len(chunks)}",
            flush=True,
        )
    encoded = encodings(brain_id)
    cut = max(1, min(int(k), len(doc_ids) or 1))
    ranked_by_qid: dict[str, list[str]] = {}
    for index, query in enumerate(queries):
        qid = str(query.get("qid") or "")
        qtext = str(query.get("query") or "")
        if encoded:
            q_toks = encode_tokens(qtext)
            scores = _maxsim_scores(q_toks, encoded)
            ranked_by_qid[qid] = rank_docs(scores, doc_ids)[:cut]
        else:
            ids, scores, _ = retrieve(qtext, brain_id, cut)
            ranked_by_qid[qid] = rank_docs(
                {str(doc_id): float(scores.get(doc_id, 0.0)) for doc_id in ids},
                doc_ids,
            )[:cut]
        if (index + 1) % 10 == 0 or index + 1 == len(queries):
            print(f"colbert retrieve {index + 1}/{len(queries)}", flush=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return ranked_by_qid, elapsed_ms, model_name()


def run_local_colbert(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    k: int = 50,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    brain_id: str | None = None,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    ranked_by_qid, encode_ms, model = retrieve_colbert(
        docs,
        queries,
        k=k,
        brain_id=brain_id or BRAIN_ID,
    )
    metrics, per_query = evaluate_dense(
        ranked_by_qid, queries, ks=ks, encode_ms=encode_ms
    )
    return {
        "status": "ok",
        "brain_id": brain_id or BRAIN_ID,
        "dataset": dataset_name,
        "fusion": "none",
        "rerank": "none",
        "channels": ["harness-colbert"],
        "rank_pool": False,
        "skip_enrichment": True,
        "k": k,
        "ks": list(ks),
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_docs_mapped": len(docs),
        "ce_model": model,
        "protocol": (
            "local ColBERT MaxSim over search_esci_74.jsonl; "
            "not fused with passages; not MiniLM ANCE; not Reddy 0.857"
        ),
        "metrics": metrics,
        "queries": per_query,
        "ingest": {"status": "completed", "n_docs": 0, "tasks": [], "reused": True},
    }
