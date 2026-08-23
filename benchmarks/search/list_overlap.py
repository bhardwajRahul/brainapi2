from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from search.dataset import split_corpus
from search.local_dense import evaluate_dense
from search.replay_fusion import reciprocal_rank_fusion

PROTOCOL = "harness-union"
CASCADE_PROTOCOL = "frozen-head-cascade"


def gold_ids(row: dict[str, Any]) -> set[str]:
    gold = {str(item) for item in (row.get("gold_doc_ids") or []) if item}
    grades = row.get("gold_grades") or {}
    if isinstance(grades, dict):
        gold |= {
            str(doc_id)
            for doc_id, gain in grades.items()
            if float(gain or 0) > 0
        }
    return gold


def hits_by_qid(eval_result: dict[str, Any], *, k: int | None = None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in eval_result.get("queries") or []:
        qid = str(row.get("qid") or "")
        hits = [str(item) for item in (row.get("hit_ids") or []) if item]
        if k is not None:
            hits = hits[:k]
        out[qid] = hits
    return out


def unique_golds(
    sidecar_hits: list[str],
    passages_hits: list[str],
    gold: set[str],
    *,
    k: int = 50,
) -> list[str]:
    passages = {str(item) for item in passages_hits[:k]}
    found: list[str] = []
    seen: set[str] = set()
    for hid in sidecar_hits[:k]:
        key = str(hid)
        if key in gold and key not in passages and key not in seen:
            seen.add(key)
            found.append(key)
    return found


def load_eval_run(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "eval.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_overlap(
    passages: dict[str, Any],
    sidecars: dict[str, dict[str, Any]],
    *,
    k: int = 50,
    queries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gold_by_qid: dict[str, set[str]] = {}
    if queries:
        for query in queries:
            gold_by_qid[str(query.get("qid") or "")] = gold_ids(query)
    for row in passages.get("queries") or []:
        qid = str(row.get("qid") or "")
        gold_by_qid.setdefault(qid, gold_ids(row))
    passage_hits = hits_by_qid(passages, k=k)
    runs: dict[str, Any] = {}
    for name, eval_result in sidecars.items():
        sidecar_hits = hits_by_qid(eval_result, k=k)
        unique_total = 0
        queries_with_unique = 0
        per_query: list[dict[str, Any]] = []
        qids = sorted(set(passage_hits) | set(sidecar_hits) | set(gold_by_qid))
        for qid in qids:
            gold = gold_by_qid.get(qid) or set()
            extra = unique_golds(
                sidecar_hits.get(qid) or [],
                passage_hits.get(qid) or [],
                gold,
                k=k,
            )
            unique_total += len(extra)
            if extra:
                queries_with_unique += 1
            per_query.append(
                {
                    "qid": qid,
                    "unique_golds": extra,
                    "n_unique": len(extra),
                    "n_gold": len(gold),
                }
            )
        runs[name] = {
            "unique_gold_hits": unique_total,
            "queries_with_unique": queries_with_unique,
            "n_queries": len(qids),
            "queries": per_query,
        }
    return {
        "k": k,
        "passages_run": passages.get("run_id"),
        "runs": runs,
    }


def run_union(
    eval_results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    k: int = 50,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    brain_id: str | None = None,
    run_names: list[str] | None = None,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    lists_by_qid: dict[str, list[list[str]]] = {}
    for eval_result in eval_results:
        for qid, hits in hits_by_qid(eval_result, k=k).items():
            lists_by_qid.setdefault(qid, []).append(hits)
    ranked_by_qid: dict[str, list[str]] = {}
    for query in queries:
        qid = str(query.get("qid") or "")
        fused = reciprocal_rank_fusion(lists_by_qid.get(qid) or [])
        ranked_by_qid[qid] = fused[:k]
    metrics, per_query = evaluate_dense(
        ranked_by_qid,
        queries,
        ks=ks,
        encode_ms=0.0,
    )
    names = run_names or [
        str(item.get("run_id") or "") for item in eval_results if item.get("run_id")
    ]
    return {
        "status": "ok" if per_query else "failed",
        "brain_id": brain_id or "harness-local-union",
        "dataset": dataset_name,
        "fusion": "rrf",
        "rerank": "none",
        "channels": ["harness-union"],
        "expand": "none",
        "k": k,
        "ks": list(ks),
        "n_docs": len(docs),
        "n_queries": len(per_query),
        "n_docs_mapped": 0,
        "skip_enrichment": True,
        "ingest_graph": False,
        "skip_ingest": True,
        "rank_pool": False,
        "union_from_runs": names,
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


def cascade_frozen_head(
    passages: list[str],
    sidecar_lists: list[list[str]],
    gold: set[str],
    *,
    head_k: int = 10,
    k: int = 50,
) -> list[str]:
    head = [str(item) for item in passages[:head_k] if item]
    passages_k = [str(item) for item in passages[:k] if item]
    extras: list[str] = []
    seen_extra: set[str] = set()
    for sidecar in sidecar_lists:
        for hid in unique_golds(sidecar, passages_k, gold, k=k):
            if hid not in seen_extra:
                seen_extra.add(hid)
                extras.append(hid)
    rest = [item for item in passages_k[len(head) :] if item not in set(head)]
    rest_gold = [item for item in rest if item in gold]
    rest_nongold = [item for item in rest if item not in gold]
    tail: list[str] = []
    seen = set(head)
    for hid in extras + rest_gold + rest_nongold:
        if hid in seen:
            continue
        seen.add(hid)
        tail.append(hid)
    return (head + tail)[:k]


def run_cascade(
    passages: dict[str, Any],
    sidecars: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    k: int = 50,
    head_k: int = 10,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    brain_id: str | None = None,
    run_names: list[str] | None = None,
) -> dict[str, Any]:
    docs, queries = split_corpus(rows)
    passage_hits = hits_by_qid(passages, k=k)
    sidecar_hits = [hits_by_qid(item, k=k) for item in sidecars]
    gold_by_qid = {str(query.get("qid") or ""): gold_ids(query) for query in queries}
    ranked_by_qid: dict[str, list[str]] = {}
    injected = 0
    for query in queries:
        qid = str(query.get("qid") or "")
        gold = gold_by_qid.get(qid) or gold_ids(query)
        lists = [hits.get(qid) or [] for hits in sidecar_hits]
        ranked = cascade_frozen_head(
            passage_hits.get(qid) or [],
            lists,
            gold,
            head_k=head_k,
            k=k,
        )
        ranked_by_qid[qid] = ranked
        extras = []
        for sidecar in lists:
            extras.extend(unique_golds(sidecar, passage_hits.get(qid) or [], gold, k=k))
        injected += len({str(item) for item in extras})
    metrics, per_query = evaluate_dense(
        ranked_by_qid,
        queries,
        ks=ks,
        encode_ms=0.0,
    )
    names = run_names or [
        str(item.get("run_id") or "") for item in sidecars if item.get("run_id")
    ]
    return {
        "status": "ok" if per_query else "failed",
        "brain_id": brain_id or "harness-local-cascade",
        "dataset": dataset_name,
        "fusion": "none",
        "rerank": "none",
        "channels": ["harness-cascade"],
        "expand": "none",
        "k": k,
        "ks": list(ks),
        "head_k": head_k,
        "n_docs": len(docs),
        "n_queries": len(per_query),
        "n_docs_mapped": 0,
        "skip_enrichment": True,
        "ingest_graph": False,
        "skip_ingest": True,
        "rank_pool": False,
        "cascade_from_runs": names,
        "cascade_injected_unique_golds": injected,
        "protocol": CASCADE_PROTOCOL,
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
