from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from search.dataset import load_records, split_corpus
from search.finetune_esci_4class import LABELS, LABEL_TO_ID, held_out_query_ids
from search.mine_retrieved_lists import is_held_out, write_lists

PROTECTED_OUT_NAMES = {
    "esci_retrieved_lists.jsonl",
    "search_esci.jsonl",
    "search_esci_74.jsonl",
}


def label_from_gain(gain: float) -> str:
    value = float(gain or 0.0)
    if value >= 0.99:
        return "E"
    if value >= 0.05:
        return "S"
    if value > 0.0:
        return "C"
    return "I"


def _chunk_to_doc(row: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for hit in row.get("hits") or []:
        chunk_id = str(hit.get("id") or "")
        doc_id = str(hit.get("doc_id") or "")
        if chunk_id and doc_id:
            mapping[chunk_id] = doc_id
    return mapping


def _as_docs(ids: list[str], mapping: dict[str, str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        key = mapping.get(str(raw), str(raw))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _queries_by_qid(queries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in queries:
        qid = str(row.get("qid") or "")
        if qid:
            out[qid] = row
    return out


def _passage_for(doc: dict[str, Any] | None, pid: str) -> str:
    if not doc:
        return pid
    text = str(doc.get("text") or "").strip()
    if text:
        return text
    title = str(doc.get("title") or "").strip()
    if title:
        return f"Title: {title}"
    return pid


def rows_from_eval(
    eval_result: dict[str, Any],
    *,
    docs: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    holdout: set[str],
    k: int = 50,
) -> list[dict[str, Any]]:
    docs_by_id = {
        str(doc.get("doc_id") or ""): doc
        for doc in docs
        if str(doc.get("doc_id") or "")
    }
    query_rows = _queries_by_qid(queries)
    cut = max(1, int(k))
    rows: list[dict[str, Any]] = []
    for row in eval_result.get("queries") or []:
        qid = str(row.get("qid") or "")
        if not qid or is_held_out(qid, holdout):
            continue
        source = query_rows.get(qid) or row
        query = str(source.get("query") or row.get("query") or "")
        grades = source.get("gold_grades") or row.get("gold_grades") or {}
        if not isinstance(grades, dict):
            grades = {}
        mapping = _chunk_to_doc(row)
        hit_ids = [str(item) for item in (row.get("hit_ids") or []) if item]
        pids = _as_docs(hit_ids, mapping)[:cut]
        for pid in pids:
            gain = float(grades.get(pid) or 0.0)
            label = label_from_gain(gain)
            if label not in LABEL_TO_ID:
                label = "I"
            rows.append(
                {
                    "query_id": qid,
                    "query": query,
                    "product_id": pid,
                    "label": label,
                    "passage": _passage_for(docs_by_id.get(pid), pid),
                }
            )
    return rows


def export_hybrid_lists(
    *,
    eval_result: dict[str, Any],
    dataset_path: Path,
    holdout_path: Path,
    out_path: Path,
    k: int = 50,
) -> dict[str, Any]:
    name = out_path.name
    if name in PROTECTED_OUT_NAMES:
        raise ValueError(f"refusing to overwrite {out_path}")
    holdout = held_out_query_ids(holdout_path)
    docs, queries = split_corpus(load_records(dataset_path))
    rows = rows_from_eval(
        eval_result,
        docs=docs,
        queries=queries,
        holdout=holdout,
        k=k,
    )
    write_lists(rows, out_path)
    counts = {label: 0 for label in LABELS}
    by_q: dict[str, list[str]] = {}
    for row in rows:
        counts[str(row["label"])] = counts.get(str(row["label"]), 0) + 1
        by_q.setdefault(str(row["query_id"]), []).append(str(row["label"]))
    n_with_i = sum(1 for labels in by_q.values() if "I" in labels)
    meta = {
        "n_queries": len(by_q),
        "n_pairs": len(rows),
        "n_holdout_qids": len(holdout),
        "k": int(k),
        "label_counts": counts,
        "n_lists_with_i": n_with_i,
        "source": "hybrid-k50",
        "out": str(out_path),
        "dataset": str(dataset_path),
        "holdout": str(holdout_path),
    }
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
